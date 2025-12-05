#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
import os, time, datetime, sys, shutil, tempfile, subprocess
import gpt as g


class lock:
    def __init__(self, root, timeout_wait=300, timeout_stale=3600, throttle_time=1e-3):
        self.lock_dir = f"{root}/.gpt_jobs_lock"
        t0 = g.time()

        # make root directory if it does not exist
        os.makedirs(root, exist_ok=True)

        g.message(f"Lock on {self.lock_dir} query")
        if os.path.exists(self.lock_dir):
            timeout = (
                datetime.datetime.now()
                - datetime.datetime.fromtimestamp(os.stat(self.lock_dir).st_ctime)
            ).total_seconds()
            if timeout > timeout_stale:
                os.rmdir(self.lock_dir)
                g.message(f"Lock {self.lock_dir} became stale, remove")

        while g.time() - t0 < timeout_wait:
            if not os.path.exists(self.lock_dir):
                try:
                    os.mkdir(self.lock_dir)
                    g.message(f"Lock acquired on {self.lock_dir}")
                    return
                except FileExistsError:
                    pass
            time.sleep(throttle_time)

        g.message(f"Lock on {self.lock_dir} expired")
        sys.exit(1)

    def __del__(self):
        os.rmdir(self.lock_dir)
        g.message(f"Lock released on {self.lock_dir}")
    

class scheduler_slurm:
    def __init__(self, env):
        self.env = env

    def get_step(self):
        return self.env["SLURM_JOB_ID"] + "." + self.env["SLURM_STEP_ID"]

    def is_step_running(self, step):
        stat = os.system(f"sstat -j {step} 2>&1 | grep -q error") != 0
        return stat



class scheduler_pbs:
    def __init__(self, env):
        self.env = env
        if g.rank() == 0:
            self.job_file = tempfile.NamedTemporaryFile(delete=True, dir=".", prefix=".gpt-job-file.")
            job_file_name = self.job_file.name
        else:
            job_file_name = ""

        self.job_file_name = g.broadcast(0, job_file_name)

        if g.rank() == 0:
            self.qstat_last_update = -1000
            self.update_qstat()

    def update_qstat(self):
        if g.time() - self.qstat_last_update < 4:
            return
        self.qstat_last_update = g.time()
        
        process = subprocess.Popen(["qstat"], stdout=subprocess.PIPE)

        self.running_jobs = []
        for line in process.stdout:
            args = [y for y in line.decode("utf-8").split(' ') if y != '']
            if len(args) > 4 and args[4] == "R":
                self.running_jobs.append(args[0].split(".")[0])
                
        process.wait()

    def get_step(self):
        return self.env["PBS_JOBID"].split(".")[0] + "|||" + self.job_file_name

    def is_step_running(self, step_all):
        step_all = step_all.split("|||")
        step = step_all[0]

        if len(step_all) > 1:
            test_job_file = step_all[1]
            if not os.path.exists(test_job_file):
                g.message(f"Scheduler PBS: job {step} is no longer running because jobfile {test_job_file} is absent")
                return False

        self.update_qstat()
            
        stat = step in self.running_jobs
        g.message(f"Scheduler PBS: job {step} is running {stat} (all: {self.running_jobs})")

        # if the job is not running but the job file is still present, clean it up
        if not stat and len(step_all) > 1:
            test_job_file = step_all[1]
            if os.path.exists(test_job_file) and g.rank() == 0:
                os.unlink(test_job_file)

        return stat


class scheduler_unknown:
    def __init__(self):
        pass

    def get_step(self):
        return "unknown"

    def is_step_running(self, step):
        return True
    

class scheduler:
    def __init__(self):
        env = dict(os.environ)
        if "SLURM_JOB_ID" in env:
            self.kernel = scheduler_slurm(env)
        elif "PBS_JOBID" in env:
            self.kernel = scheduler_pbs(env)
        else:
            self.kernel = scheduler_unknown()

    def get_step(self):
        return self.kernel.get_step()

    def is_step_running(self, step):
        return self.kernel.is_step_running(step)


scd = None


class base:
    def __init__(self, name, needs):
        global scd
        self.name = name
        self.needs = needs
        self.weight = 1.0
        if scd is None:
            scd = scheduler()

    def perform(self, root):
        raise NotImplementedError(f"{self.name} perform not implemented")

    def check(self, root):
        raise NotImplementedError(f"{self.name} perform not implemented")

    def has_started(self, root):
        return os.path.exists(f"{root}/{self.name}")

    def is_running(self, root):
        if not os.path.exists(f"{root}/{self.name}/.started"):
            return False
        step = open(f"{root}/{self.name}/.started").read().split("\n")
        if len(step) < 3:
            # This can only occur if .started file was corrupted, e.g., via lacking disk space.
            # Assume job is no longer running.
            return False
        step = step[2]
        return scd.is_step_running(step)
    
    def has_failed(self, root):
        return self.has_started(root) and not self.has_completed(root) and not self.is_running(root)

    def run_time(self, root):
        return (
            datetime.datetime.now()
            - datetime.datetime.fromtimestamp(os.stat(f"{root}/{self.name}").st_ctime)
        ).total_seconds()

    def purge(self, root):
        if g.rank() != 0:
            return
        
        # if directory has structure of a job directory, purge (sanity check)
        if os.path.exists(f"{root}/{self.name}/.started"):
            shutil.rmtree(f"{root}/{self.name}")
            g.message(f"Purged {root}/{self.name}")
        else:
            # otherwise only remove if empty
            try:
                os.rmdir(f"{root}/{self.name}")
            except OSError:
                g.message("Directory was not empty")

    def atomic_reserve_start(self, root):

        lock_dir = f"{root}/{self.name}"
        
        # first create parent directory in non-atomic manner
        os.makedirs(os.path.dirname(lock_dir), exist_ok=True)

        # then create final directory in atomic manner
        try:
            os.mkdir(lock_dir)
            return True
        except FileExistsError:
            return False

    def has_completed(self, root):
        fd = f"{root}/{self.name}"
        if os.path.exists(f"{fd}/.checked"):
            return True
        if os.path.exists(f"{fd}/.completed"):
            if self.check(root):
                f = open(f"{fd}/.checked", "wt")
                f.write(time.asctime() + "\n")
                f.close()
                return True
                
        return False

    def __call__(self, root):
        fd = f"{root}/{self.name}"

        if g.rank() == 0:
            os.makedirs(fd, exist_ok=True)
            f = open(f"{fd}/.started", "wt")
            f.write(time.asctime() + "\n")
            f.write(str(sys.argv) + "\n")
            f.write(scd.get_step() + "\n")
            f.close()

        g.barrier()

        self.perform(root)

        if g.rank() == 0:
            f = open(f"{fd}/.completed", "wt")
            f.write(time.asctime() + "\n")
            f.close()

        g.barrier()


def get_next_name(root, jobs, max_weight, stale_seconds):
    # create lut
    lut = {}
    for j in jobs:
        lut[j.name] = j

    for j in jobs:
        if max_weight is None or j.weight <= max_weight:
            has_started = j.has_started(root)
            if has_started:
                if j.has_failed(root):
                    g.message(f"Job {j.name} has failed; purge")
                    j.purge(root)
                    has_started = False
            if has_started and stale_seconds is not None:
                if not j.has_completed(root):
                    run_time = j.run_time(root)
                    if run_time > stale_seconds:
                        g.message(f"Job {j.name} is stale after {run_time} seconds; purge")
                        j.purge(root)
                        has_started = False

            if not has_started:
                # check dependencies
                dependencies_ok = True
                for dep_j in [lut[d] for d in j.needs]:
                    if not dep_j.has_completed(root):
                        dependencies_ok = False
                        g.message(f"Dependency {dep_j.name} of {j.name} is not yet satisfied.")
                        break
                if dependencies_ok:
                    # last check if in meantime somebody else has started running same job
                    if j.atomic_reserve_start(root):
                        return j.name

    return ""


def next(root, jobs, max_weight=None, stale_seconds=None):
    if g.rank() == 0:
        l = lock(root)
        j = get_next_name(root, jobs, max_weight, stale_seconds).encode("utf-8")
        del l
    else:
        j = bytes()

    j_name = g.broadcast(0, j).decode("utf-8")
    for j in jobs:
        if j.name == j_name:
            g.message(
                f"""
--------------------------------------------------------------------------------
   Start job {j.name}
--------------------------------------------------------------------------------
"""
            )
            t0 = g.time()
            j(root)
            t1 = g.time()
            g.message(
                f"""
--------------------------------------------------------------------------------
   Completed {j.name} in {t1-t0} seconds
--------------------------------------------------------------------------------
"""
            )
            return j
    return None
