Option Explicit

Dim shell, fso, root, pythonw, command, arg

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName)

pythonw = root & "\venv\Scripts\pythonw.exe"
If Not fso.FileExists(pythonw) Then
  pythonw = "pythonw.exe"
End If

shell.CurrentDirectory = root
command = Quote(pythonw) & " " & Quote(root & "\launch_gradio.py") & " --autolaunch"
For Each arg In WScript.Arguments
  command = command & " " & Quote(CStr(arg))
Next

shell.Run command, 0, False

Function Quote(value)
  Quote = """" & Replace(value, """", """""") & """"
End Function
