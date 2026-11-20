import ctypes, os, torch
_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libmtltrace.dylib"))
_lib.mtltrace_install()
def start(path): torch.mps.synchronize(); _lib.mtltrace_start(path.encode())
def stop(): torch.mps.synchronize(); _lib.mtltrace_stop()
def mark(s): _lib.mtltrace_mark(str(s).encode())
