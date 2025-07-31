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
import gpt, sys, os, signal, datetime, socket, cgpt
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

def setup():
    global beats, pid, t0
    
    signal.signal(signal.SIGUSR2, backtrace_signal_handler)

    # signal heartbeats for machines that are prone to hangs
    if gpt.default.has("--signal-heartbeat"):
        import time

        bpm = gpt.default.get_int("--signal-heartbeat-bpm", 1)
        beats = 0

        pid = os.fork()

        gpt.message(f"Heartbeat setup with {bpm} bpm, pid = {pid} is here")
        
        if pid == 0:

            # wait for 30 seconds before starting
            time.sleep(30)
        
            # I am the monitor job.  Send bpm heartbeats to main process
            # and make sure it is still responding every minute.  If not,
            # first kill, then terminate it.
            parentid = os.getppid()
            t0 = cgpt.time()
        
            def signal_handler_monitor(sig, frame):
                global t0
                t0 = cgpt.time()
            
            signal.signal(signal.SIGUSR1, signal_handler_monitor)

            sig_state = 0
            while True:
                try:
                    t = cgpt.time()
                    if t - t0 > 60*4 and sig_state == 0:
                        os.write(sys.stderr.fileno(), f"Process {parentid} on {socket.gethostname()} froze, send BACKTRACE signal\n".encode("utf-8"))
                        os.kill(parentid, signal.SIGUSR2)
                        sig_state = 1
                    elif t - t0 > 60*5 and sig_state == 1:
                        os.write(sys.stderr.fileno(), f"Process {parentid} on {socket.gethostname()} froze, send KILL signal\n".encode("utf-8"))
                        os.kill(parentid, signal.SIGKILL)
                        sig_state = 2
                    elif t - t0 > 60*6 and sig_state == 2:
                        os.write(sys.stderr.fileno(), f"Process {parentid} on {socket.gethostname()} froze, send TERM signal\n".encode("utf-8"))
                        os.kill(parentid, signal.SIGTERM)
                    else:
                        os.kill(parentid, signal.SIGUSR1)
                except ProcessLookupError:
                    sys.exit(0)
                time.sleep(60 / bpm)

        else:

            def signal_handler_noop(sig, frame):
                global beats, pid
                beats += 1
                if beats >= bpm:
                    if gpt.rank() == 0:
                        # report heartbeats every minite to stdout
                        msg = f"GPT : {gpt.time():14.6f} s : {beats} heartbeat(s) received, send signal to {pid}\n"
                        os.write(sys.stdout.fileno(), msg.encode("utf-8"))
                    # and tell the monitor that we are still alive
                    os.kill(pid, signal.SIGUSR1)
                    beats = 0

            signal.signal(signal.SIGUSR1, signal_handler_noop)


    # monitor all signals and respond with backtrace
    if gpt.default.is_verbose("all_signals_backtrace"):
        for s in [
                signal.SIGBUS,
                signal.SIGFPE,
                signal.SIGHUP,
                signal.SIGINT,
                signal.SIGTERM,
                signal.SIGSEGV,
        ]:
            signal.signal(s, backtrace_signal_handler)
            
        import ctypes

        c_globals = ctypes.CDLL(None)

        @ctypes.CFUNCTYPE(None, ctypes.c_int)
        def sigabrt_handler(sig):
            backtrace_signal_handler(sig, sys._getframe(0))

        c_globals.signal(signal.SIGABRT, sigabrt_handler)
