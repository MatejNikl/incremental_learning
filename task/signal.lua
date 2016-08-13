local sig = require "posix.signal"

local function signal_handler(signo)
    print("Received signal: " .. signo)
    if signo == sig.SIGINT then
        _G.interrupted = true
    end
end
sig.signal_handler = signal_handler

return sig
