local cmdio = {}

local function prompt(string)
    io.write(string)
    local read = io.read()
    return #read ~= 0 and read or nil
end
cmdio.prompt = prompt

local function check_useragrees(question)
    local ans

    repeat
        ans = prompt(question  .. " ((y)es/(n)o)? ")
    until ans == "y" or ans == "yes" or ans == "n" or ans == "no"

    return ans == "y" or ans == "yes"
end
cmdio.check_useragrees = check_useragrees

--function dissect_path(path)
--    local dir, fname, ext = string.match(path, "(.*/)([^/]+)(%.[^/.]+)$")
--    return dir, fname, ext
--end
--hlp.dissect_path = dissect_path

return cmdio
