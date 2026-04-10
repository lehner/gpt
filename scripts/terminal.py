#!/usr/bin/env python3
import sys
import os
import time
from curses import wrapper
import curses
import curses.textpad

stdout=sys.argv[1]

f = open(stdout, "rt")

lines = []
a = None
def update_stdout():
    global lines, f, a
    while True:
        line = f.readline()
        if line == "":
            break

        if "<TERMINAL>" in line:
            a = line.split("<TERMINAL>")
            assert len(a) == 2
            a = a[1].split("</TERMINAL>")
            assert len(a) == 2
            a = a[0]
            if not os.path.exists(a):
                os.makedirs(a, exist_ok=True)
        else:
            lines.append(line.rstrip("\n"))


def main(stdscr):
    stdscr.clear()

    H, W = stdscr.getmaxyx()

    cmd = ""
    cmds = []

    offset = 0
    cmd_idx = -1
    editing = True
    cursor = 0
    last_box = ""
    title = f"GPT terminal connected to {stdout}"
    helpstr = "ctrl + N for multi-line input, ctrl + O to continue editing, ctrl + G to submit it"
    stdscr.addstr(1, (W - len(title)) // 2, title)
    stdscr.addstr(2, (W - len(helpstr)) // 2, helpstr)
    stdscr.addstr(3, 0, "-" * W, curses.color_pair(2))
    stdscr.nodelay(True)
    stdscr.addstr(H - 3, 0, "-" * W, curses.color_pair(2))
    stdscr.keypad(True)

    curses.start_color()
    curses.use_default_colors()

    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)
    
    L = H - 7
    assert L > 0
    
    while True:
        update_stdout()

        stdscr.addstr(3, W//2 - 2, "[" + str(offset) + "]" + "-" * 10, curses.color_pair(2))

        for j in range(min(L, len(lines) - offset)):
            ln = lines[-j-1 - offset].replace("\t", " "*4)
            stdscr.addstr(L + 3 - j, 0, ln + " "*(W-len(ln)))
        for j in range(min(L, len(lines) - offset), L):
            stdscr.addstr(L + 3 - j, 0, "*" + " "*(W-1))

        stdscr.refresh()
        c = stdscr.getch()
        if c != -1:
            if c == curses.KEY_PPAGE:
                if offset < len(lines) - 10:
                    offset += 10
            elif c == curses.KEY_NPAGE:
                if offset >= 10:
                    offset -= 10
            elif c == curses.KEY_END:
                offset = 0    
            elif c == curses.KEY_HOME:
                if len(lines) > 10:
                    offset = (len(lines) - 10) // 10 * 10

            if a is not None:

                if c == 127 or c == 263: # backspace
                    if cursor > 0:
                        cmd = cmd[0:cursor-1] + cmd[cursor:]
                        cursor -= 1
                    editing = True
                elif c == 1: # ctrl+A
                    cursor = 0
                elif c == 5: # ctrl+E
                    cursor = len(cmd)
                elif c == 14 or c == 15: # ctrl+N or ctrl+O
                    win = curses.newwin(L, W, 4, 0)
                    win.clear()
                    if c == 15:
                        for l, ln in enumerate(last_box.split("\n")):
                            win.addstr(l, 0, ln)
                    x = curses.textpad.Textbox(win)
                    y = x.edit()
                    last_box = y

                    g = open(f"{a}/command", "wt")
                    g.write(y + "\n")
                    g.close()
                elif c == 10: # enter
                    g = open(f"{a}/command", "wt")
                    g.write(cmd + "\n")
                    g.close()
                    cmds.append(cmd)
                    cmd_idx = len(cmds) - 1
                    cmd = ""
                    cursor = 0
                    editing = True
                elif c < 127:
                    cmd = cmd[0:cursor] + chr(c) + cmd[cursor:]
                    cursor += 1
                    editing = True
                elif c == curses.KEY_RIGHT:
                    if cursor < len(cmd):
                        cursor += 1
                elif c == curses.KEY_LEFT:
                    if cursor > 0:
                        cursor -= 1
                elif c == curses.KEY_UP:
                    if len(cmds) > 0:
                        if editing:
                            cmds.append(cmd)
                            editing = False
                        cmd = cmds[cmd_idx]
                        cursor = 0
                        if cmd_idx > 0:
                            cmd_idx -= 1
                        else:
                            cmd_idx = len(cmds) - 1
                elif c == curses.KEY_DOWN:
                    if len(cmds) > 0:
                        if editing:
                            cmds.append(cmd)
                            editing = False
                        cmd = cmds[cmd_idx]
                        cursor = 0
                        if cmd_idx < len(cmds) - 1:
                            cmd_idx += 1
                        else:
                            cmd_idx = 0

        stdscr.move(H - 2, 3 + cursor)
        stdscr.refresh()
        curses.curs_set(1)
        if a is not None:
            stdscr.addstr(H - 2, 0, " > " + cmd + " " * (W - len(cmd) - 3))
        else:
            stdscr.addstr(H - 2, 0, " > waiting for terminal setup to complete (takes up to 60 seconds)")
        time.sleep(0.001)

try:
    wrapper(main)
except KeyboardInterrupt:
    pass

if a is not None:
    try:
        os.rmdir(a)
    except FileNotFoundError:
        pass
