Option Explicit

Dim shell, fso, root, command, arg

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName)

shell.CurrentDirectory = root
command = "powershell -NoProfile -ExecutionPolicy Bypass -File " & Quote(root & "\scripts\install_aiwf_studio.ps1")

For Each arg In WScript.Arguments
  command = command & " " & Quote(CStr(arg))
Next

shell.Run command, 0, False

Function Quote(value)
  Quote = """" & Replace(value, """", """""") & """"
End Function
