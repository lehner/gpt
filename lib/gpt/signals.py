#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2025  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import gpt, sys, os, signal, datetime, socket, cgpt, threading, time, ctypes, multiprocessing
from inspect import getframeinfo, stack


# allow backtraces to be triggered by a signal
def backtrace_signal_handler(sig, frame):

    now = datetime.datetime.now()
    log_directory = "log/" + now.strftime("%Y-%m-%d")
    if not os.path.exists(log_directory):
        os.makedirs(log_directory, exist_ok=True)

    log_filename = f"{log_directory}/backtrace.{gpt.rank()}." + now.strftime("%H-%M-%f")
    sys.stderr.write(f"Requested GPT backtrace {sig}; saved in {log_filename}\n")
    sys.stderr.flush()

    fout = open(log_filename, "wt")
    fout.write(f"Host: {socket.gethostname()}\n")

    while frame is not None:
        caller = getframeinfo(frame)
        fout.write(f"{caller.filename}:{caller.lineno}\n")
        frame = frame.f_back

    fout.close()


beats = None
pid = None
t0 = None
last_alive_t = None

# determine if pid is in a syscall
def get_syscall(pid):
    try:
        with open(f"/proc/{pid}/syscall") as f:
            return f.read().strip()
    except:
        return None




def setup():
    global beats, pid, t0, last_alive_t, global_aborter

    signal.signal(signal.SIGUSR2, backtrace_signal_handler)

    # signal heartbeats for machines that are prone to hangs
    if gpt.default.has("--signal-heartbeat"):

        bpm = gpt.default.get_int("--signal-heartbeat-bpm", 1)
        beats = 0

        last_alive_t = multiprocessing.Value('d', 0)
        pid = os.fork()

        gpt.message(f"Heartbeat setup with {bpm} bpm, pid = {pid} is here")
        
        if pid == 0:

            # wait for 10 seconds before starting
            time.sleep(10)
        
            # I am the monitor job.  See if other job is stuck in syscall.
            parentid = os.getppid()
        
            sig_state = 0
            sc_last = ""
            last_alive_t.value = cgpt.time()

            def abort():
                os.write(sys.stdout.fileno(), f"{gpt.rank()} pid={parentid} ABORT\n".encode("utf-8"))
                try:
                    # os.kill(parentid, signal.SIGKILL)
                    os.kill(parentid, signal.SIGABRT)

                    # this is crude and should be improved!
                    os.system("killall python3")
                except ProcessLookupError:
                    pass
                sys.exit(1)
                
            while True:
                t1 = cgpt.time()
                if t1 - last_alive_t.value > 300 and sig_state == 1:
                    # keep track of how many signals we have sent and give up/abort job after a while
                    # if job does not respond, we may need to find another way to abort it; maybe send signals to all other
                    # ranks on same node to kill?
                    abort()

                elif t1 - last_alive_t.value > 120 and sig_state == 0:
                    sc = get_syscall(parentid)
                    if sc is None:
                        # sc = "dead"
                        abort()
                    if sc.split(" ")[0].strip() == "-1":
                        sc = "running"
                    os.write(sys.stdout.fileno(), f"{gpt.rank()} {sc} pid={parentid} syscall warning last_alive_dt={cgpt.time() - last_alive_t.value}\n".encode("utf-8"))

                    # send backtrace signal both GPTs and Grids to document where we are; 
                    os.system(f"module load gdb ; echo -e \"bt\ndetatch\nquit\" | gdb -p {parentid}")
                    os.kill(parentid, signal.SIGUSR2)
                    os.kill(parentid, signal.SIGHUP)
                    sig_state = 1

                time.sleep(60 / bpm)

        else:

            # send heartbeats to the monitor job to indicate that we are still alive,
            # if the monitor job stops receiving heartbeats, it should investigate
            def i_am_alive():
                global last_alive_t
                while True:
                    time.sleep(60 / bpm)
                    last_alive_t.value = cgpt.time()

            # in CPython with a GIL, this does exactly what we want, i.e., queues
            # a call to i_am_alive in the main interpreter; if it is stuck, the monitor
            # will notice; without the GIL this would just keep running...
            thread = threading.Thread(target=i_am_alive, args=(), daemon=True)
            thread.start()


    # monitor all signals and respond with backtrace
    if gpt.default.is_verbose("all_signals_backtrace"):
        for s in [
                signal.SIGBUS,
                signal.SIGFPE,
                # signal.SIGHUP,  <-- leave this for Grid
                signal.SIGINT,
                signal.SIGTERM,
                signal.SIGSEGV,
        ]:
            signal.signal(s, backtrace_signal_handler)
            
        c_globals = ctypes.CDLL(None)

        @ctypes.CFUNCTYPE(None, ctypes.c_int)
        def sigabrt_handler(sig):
            backtrace_signal_handler(sig, sys._getframe(0))

        c_globals.signal(signal.SIGABRT, sigabrt_handler)

    # if one rank exits, trigger mpi abort (mpich workaround)
    if gpt.default.has("--abort-on-exit"):

        class aborter:
            def __init__(self):
                pass

        def __del__(self):
            gpt.abort()

        global global_aborter
        global_aborter = aborter()
