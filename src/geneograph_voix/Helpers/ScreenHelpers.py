import re
import sys

import tkinter as tk

from typing import Union

def _set_windows_dpi_awareness():
    try:
        import ctypes
        if sys.platform.startswith("win"):
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                try:
                    ctypes.windll.user32.SetProcessDPIAware()
                except Exception:
                    pass
    except Exception:
        pass

# --- UI Helpers - High-DPI fitting + main window size restore ---
def _parse_geometry(geom: str):
    try:
        m = re.match(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", str(geom))
        if not m:
            return None
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    except Exception:
        return None

def _get_workarea_bounds(root):
    """
    Return work-area bounds in **Tk units**: (work_w, work_h, work_x, work_y).
    On Windows we query the monitor work area in physical pixels and convert
    to Tk units using a per-monitor ratio derived from rcMonitor vs Tk's screen size.
    """
    # Non-Windows: fall back to Tk's screen size
    if not sys.platform.startswith("win"):
        try:
            return root.winfo_screenwidth(), root.winfo_screenheight(), 0, 0
        except Exception:
            return 1920, 1080, 0, 0

    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32

        # Structures
        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                        ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

        class MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_ulong),
                        ("rcMonitor", RECT),
                        ("rcWork", RECT),
                        ("dwFlags", ctypes.c_ulong)]

        MONITOR_DEFAULTTONEAREST = 0x00000002

        hwnd = root.winfo_id()
        hmon = user32.MonitorFromWindow(wintypes.HWND(hwnd), MONITOR_DEFAULTTONEAREST)
        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        user32.GetMonitorInfoW(hmon, ctypes.byref(mi))

        # Physical pixel rectangles
        rcM = mi.rcMonitor
        rcW = mi.rcWork
        mon_w = rcM.right - rcM.left
        mon_h = rcM.bottom - rcM.top
        work_w_px = rcW.right - rcW.left
        work_h_px = rcW.bottom - rcW.top

        # Tk reports its own screen width/height in Tk units (scaled)
        tk_sw = max(1, int(root.winfo_screenwidth()))
        tk_sh = max(1, int(root.winfo_screenheight()))

        # Convert physical → Tk units via per-axis ratios
        # (Works for per-monitor DPI; we only clamp against the monitor the window is on.)
        ratio_x = tk_sw / float(mon_w) if mon_w > 0 else 1.0
        ratio_y = tk_sh / float(mon_h) if mon_h > 0 else 1.0

        work_w = int(round(work_w_px * ratio_x))
        work_h = int(round(work_h_px * ratio_y))
        work_x = int(round(rcW.left * ratio_x))
        work_y = int(round(rcW.top * ratio_y))

        # Guard rails
        work_w = max(320, min(work_w, tk_sw))
        work_h = max(240, min(work_h, tk_sh))

        return work_w, work_h, work_x, work_y
    except Exception:
        # Fallback if anything fails
        try:
            return root.winfo_screenwidth(), root.winfo_screenheight(), 0, 0
        except Exception:
            return 1920, 1080, 0, 0

def _fit_to_screen_bounds(root, w, h, x, y):
    """
    Clamp a geometry (in Tk units) to the current monitor work area (also in Tk units).
    """
    work_w, work_h, work_x, work_y = _get_workarea_bounds(root)
    min_w, min_h = 640, 480

    w = max(min_w, min(w, work_w))
    h = max(min_h, min(h, work_h))

    # Keep the window fully inside work area
    x = max(work_x, min(x, work_x + work_w - w))
    y = max(work_y, min(y, work_y + work_h - h))
    return w, h, x, y

def _fit_to_screen(win: Union[tk.Tk, tk.Toplevel], margin: int = 60):
    """
    Make the window big enough to show all content (up to screen - margin)
    and center it. Uses *required* size so bottom buttons aren't cut off.
    """
    try:
        win.update_idletasks()
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()

        # Use the larger of current size and required (natural) size
        cur_w = win.winfo_width() or 0
        cur_h = win.winfo_height() or 0
        req_w = win.winfo_reqwidth() or cur_w or 800
        req_h = win.winfo_reqheight() or cur_h or 600

        w = max(cur_w, req_w)
        h = max(cur_h, req_h)

        # Clamp to screen with margin
        w = max(100, min(int(w), sw - margin))
        h = max(100, min(int(h), sh - margin))

        # Center
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2)
        win.geometry(f"{w}x{h}+{x}+{y}")
    except Exception:
        pass
