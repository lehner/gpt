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
import gpt as g
import signal, os, sys, time

terminal_root = g.default.get("--terminal", None)

if terminal_root is not None:

    def get_command(cmd):
        if g.rank() != 0:
            return ""
        
        # wait for 10 minutes for a command, if not, resume job
        for i in range(6000):
            time.sleep(0.1)
            if os.path.exists(cmd):
                command = open(cmd).read().strip("\n")
                os.unlink(cmd)
                return command

        return "exit"

    terminal_initialized = False
    
    def terminal_handler(signum, frame):
        global terminal_initialized

        if not terminal_initialized:
            g.message(f"<TERMINAL>{terminal_root}</TERMINAL>")
            terminal_initialized = True
            
        if os.path.exists(terminal_root):

            # security: only accept if directory was created by current user
            if os.geteuid() != os.stat(terminal_root).st_uid:
                g.message("UserID mismatch!")
                return

            signal.setitimer(
                signal.ITIMER_REAL,
                0.0,
                0.0
            )

            cmd = f"{terminal_root}/command"
            g.message("********************************************************************************")
            g.message("  Entering terminal mode")
            g.message(f"  Put command in {cmd}  (special: exit, bt)")
            g.message("")

            glb = globals()
            while True:
                command = get_command(cmd)

                g.message(f"> {command}")
                command = g.broadcast(0, command)

                if command == "exit":
                    g.message("********************************************************************************")

                    signal.setitimer(
                        signal.ITIMER_REAL,
                        60.0,
                        60.0
                    )

                    return
                elif command == "bt":
                    frame = sys._getframe(1)
                    g.message("Requested backtrace:")
                    while frame is not None:
                        g.message(f"{frame.f_code.co_filename}:{frame.f_lineno}")
                        frame = frame.f_back
                        
                else:
                    try:
                        exec(command, glb)
                        sys.stdout.flush()
                        sys.stderr.flush()
                    except Exception as inst:
                        g.message(f"Exception {inst}")

    signal.signal(signal.SIGALRM, terminal_handler)

    signal.setitimer(
        signal.ITIMER_REAL,
        60.0,
        60.0
    )
