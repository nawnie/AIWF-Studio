Unicode true
RequestExecutionLevel user
ManifestSupportedOS all
SetCompressor /SOLID zlib

!include "MUI2.nsh"
!include "nsDialogs.nsh"
!include "LogicLib.nsh"
!include "FileFunc.nsh"
!include "x64.nsh"
!include "WinVer.nsh"
!include "${__FILEDIR__}\SystemRequirements.nsh"

!ifndef PRODUCT
!define PRODUCT "AES Studio"
!endif
!ifndef OUTPUT
!error "Define OUTPUT as the path of the new installer."
!endif
!ifndef STAGE
!error "Define STAGE as the prepared payload directory."
!endif
!ifndef CREDENTIAL_BRIDGE
!define CREDENTIAL_BRIDGE "${__FILEDIR__}\AesCredentialBridge.dll"
!endif
!define REGKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT}"
Name "${PRODUCT}"
OutFile "${OUTPUT}"
InstallDir "$LOCALAPPDATA\Programs\${PRODUCT}"
InstallDirRegKey HKCU "${REGKEY}" "InstallLocation"
BrandingText "AES Studio preview"
VIProductVersion "0.1.0.0"
VIAddVersionKey "ProductName" "AES Studio"
VIAddVersionKey "FileDescription" "AES Studio Windows preview setup"
VIAddVersionKey "FileVersion" "0.1.0.0"
VIAddVersionKey "ProductVersion" "0.1.0 preview"
VIAddVersionKey "LegalCopyright" "Copyright 2026 AIWF Studio contributors"

Var Backend
Var ChoiceCpp
Var ChoiceBoth
Var HfBox
Var CivitaiBox
Var HfValue
Var CivitaiValue
Var SettingsPolicy

!define MUI_ABORTWARNING
!define MUI_WELCOMEPAGE_TITLE "Install ${PRODUCT}"
!define MUI_WELCOMEPAGE_TEXT "This per-user preview installs beside AIWF Studio. It includes the local API and NVIDIA CUDA C++ image runtime. No model is downloaded automatically.$\r$\n$\r$\nThe first run scans hardware and helps you choose your folders and a starting model."
!insertmacro MUI_PAGE_WELCOME
Page custom ModePage ModeLeave
!define MUI_PAGE_CUSTOMFUNCTION_LEAVE DirectoryLeave
!insertmacro MUI_PAGE_DIRECTORY
Page custom KeysPage KeysLeave
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_RUN "$INSTDIR\native\AESStudio.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Launch ${PRODUCT}"
!define MUI_FINISHPAGE_TEXT "${PRODUCT} has been installed. The first-run wizard checks your computer and helps you select a model."
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Function .onInit
  StrCpy $0 0
  ${If} ${RunningX64}
    StrCpy $0 1
  ${EndIf}
  ${WinVerGetBuild} $1
  !insertmacro AES_RequirementStatus $2 $0 $1
  ${If} $2 == "architecture"
    SetErrorLevel 1
    MessageBox MB_ICONSTOP|MB_OK "AES Studio requires 64-bit Windows. This PC is running 32-bit Windows." /SD IDOK
    Quit
  ${ElseIf} $2 == "version"
    SetErrorLevel 1
    MessageBox MB_ICONSTOP|MB_OK "AES Studio requires Windows 10 version 2004 (build 19041) or newer. Update Windows before installing." /SD IDOK
    Quit
  ${EndIf}
  StrCpy $Backend "both"
  ; A silent install has no interactive selection; preserve an existing profile.
  StrCpy $SettingsPolicy "apply"
  IfSilent 0 +2
    StrCpy $SettingsPolicy "preserve"
  ReadRegStr $0 HKCU "${REGKEY}" "InstallLocation"
  ${If} $0 != ""
    MessageBox MB_ICONEXCLAMATION|MB_OK "${PRODUCT} is already registered at $0. This preview does not overwrite existing installations. Uninstall it first, or keep the existing copy." /SD IDOK
    SetErrorLevel 1
    Abort
  ${EndIf}
FunctionEnd

Function DirectoryLeave
  ${GetRoot} "$INSTDIR" $0
  StrCmp "$0" "" bad_directory
  StrCmp "$INSTDIR" "$0" bad_directory
  StrCmp "$INSTDIR" "$0\" bad_directory
  ${DirState} "$INSTDIR" $1
  StrCmp $1 1 bad_directory directory_ok
bad_directory:
  MessageBox MB_ICONSTOP|MB_OK "Choose an empty, non-root installation folder. AES Studio will not overwrite an existing folder."
  Abort
directory_ok:
FunctionEnd

Function ModePage
  !insertmacro MUI_HEADER_TEXT "Choose setup" "Choose the image engines Studio will offer."
  nsDialogs::Create 1018
  Pop $0
  ${If} $0 == error
    Abort
  ${EndIf}
  ${NSD_CreateLabel} 0 0 100% 22u "On reinstall, your chosen mode updates the saved backend. Other AES settings are kept."
  Pop $0
  ${NSD_CreateFirstRadioButton} 0 24u 100% 16u "Recommended: C++ and optional Python image engines"
  Pop $ChoiceBoth
  ${NSD_CreateAdditionalRadioButton} 0 45u 100% 16u "Manual: bundled C++ image engine only (NVIDIA CUDA 13)"
  Pop $ChoiceCpp
  ${NSD_CreateLabel} 0 72u 100% 44u "The C++ engine is included. Python inference packages require separate manual installation before Python models can run. A compatible model is required for image generation."
  Pop $0
  ${If} $Backend == "cpp"
    ${NSD_Check} $ChoiceCpp
  ${Else}
    ${NSD_Check} $ChoiceBoth
  ${EndIf}
  nsDialogs::Show
FunctionEnd

Function ModeLeave
  ${NSD_GetState} $ChoiceCpp $0
  ${If} $0 == ${BST_CHECKED}
    StrCpy $Backend "cpp"
  ${Else}
    StrCpy $Backend "both"
  ${EndIf}
FunctionEnd

Function KeysPage
  !insertmacro MUI_HEADER_TEXT "Optional model provider keys" "You can skip this step and add keys later."
  nsDialogs::Create 1018
  Pop $0
  ${If} $0 == error
    Abort
  ${EndIf}
  ${NSD_CreateLabel} 0 0 100% 32u "Keys are stored in Windows Credential Manager for your Windows account. They are not checked until you use the provider. Public models usually need no key."
  Pop $0
  ${NSD_CreateLabel} 0 40u 100% 12u "Hugging Face read token (optional)"
  Pop $0
  ${NSD_CreatePassword} 0 54u 100% 14u "$HfValue"
  Pop $HfBox
  ${NSD_CreateLabel} 0 80u 100% 12u "Civitai API key (optional)"
  Pop $0
  ${NSD_CreatePassword} 0 94u 100% 14u "$CivitaiValue"
  Pop $CivitaiBox
  ${NSD_CreateLabel} 0 125u 100% 28u "Existing AES Studio credentials remain unchanged when these fields are blank."
  Pop $0
  nsDialogs::Show
FunctionEnd

Function KeysLeave
  ${NSD_GetText} $HfBox $HfValue
  ${NSD_GetText} $CivitaiBox $CivitaiValue
FunctionEnd

Section "Install"
  SetShellVarContext current
  ${GetRoot} "$INSTDIR" $0
  StrCmp "$0" "" bad_directory
  StrCmp "$INSTDIR" "$0" bad_directory
  StrCmp "$INSTDIR" "$0\" bad_directory
  ${DirState} "$INSTDIR" $1
  StrCmp $1 1 bad_directory install_ok
bad_directory:
  MessageBox MB_ICONSTOP|MB_OK "Choose an empty, non-root installation folder. AES Studio will not overwrite an existing folder." /SD IDOK
  SetErrorLevel 1
  Abort
install_ok:
  SetOutPath "$INSTDIR"
  File /r "${STAGE}\*"
  DetailPrint "Preparing the bundled offline runtime..."
  nsExec::ExecToLog '"$INSTDIR\runtime\python\python.exe" "$INSTDIR\packaging\finalize_install.py" "$INSTDIR" "$Backend" "$SettingsPolicy"'
  Pop $0
  ${If} $0 != 0
    MessageBox MB_ICONSTOP|MB_OK "AES Studio could not prepare its offline runtime (code $0). Installation is incomplete; the installation folder was preserved for diagnosis." /SD IDOK
    SetErrorLevel 1
    Abort
  ${EndIf}
  InitPluginsDir
  SetOutPath "$PLUGINSDIR"
  File /oname=AesCredentialBridge.dll "${CREDENTIAL_BRIDGE}"
  ${If} $HfValue != ""
    StrCpy $1 $HfValue
    System::Call '"$PLUGINSDIR\AesCredentialBridge.dll"::SaveKey(w "huggingface", w r1) i.r0'
    StrCpy $1 ""
    StrCpy $HfValue ""
    ${If} $0 != 1
      MessageBox MB_ICONEXCLAMATION|MB_OK "Windows could not save the Hugging Face key. Add it later from Configure Studio." /SD IDOK
    ${EndIf}
  ${EndIf}
  ${If} $CivitaiValue != ""
    StrCpy $1 $CivitaiValue
    System::Call '"$PLUGINSDIR\AesCredentialBridge.dll"::SaveKey(w "civitai", w r1) i.r0'
    StrCpy $1 ""
    StrCpy $CivitaiValue ""
    ${If} $0 != 1
      MessageBox MB_ICONEXCLAMATION|MB_OK "Windows could not save the Civitai key. Add it later from Configure Studio." /SD IDOK
    ${EndIf}
  ${EndIf}
  SetOutPath "$INSTDIR"
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  CreateDirectory "$SMPROGRAMS\${PRODUCT}"
  CreateShortcut "$SMPROGRAMS\${PRODUCT}\${PRODUCT}.lnk" "$INSTDIR\native\AESStudio.exe" "" "$INSTDIR\native\AESStudio.exe"
  CreateShortcut "$DESKTOP\${PRODUCT}.lnk" "$INSTDIR\native\AESStudio.exe" "" "$INSTDIR\native\AESStudio.exe"
  WriteRegStr HKCU "${REGKEY}" "DisplayName" "${PRODUCT}"
  WriteRegStr HKCU "${REGKEY}" "DisplayVersion" "0.1.0 preview"
  WriteRegStr HKCU "${REGKEY}" "Publisher" "Ai Embedded Systems"
  WriteRegStr HKCU "${REGKEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${REGKEY}" "DisplayIcon" "$INSTDIR\native\AESStudio.exe"
  WriteRegStr HKCU "${REGKEY}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegDWORD HKCU "${REGKEY}" "NoModify" 1
  WriteRegDWORD HKCU "${REGKEY}" "NoRepair" 1
SectionEnd

Section "Uninstall"
  SetShellVarContext current
  InitPluginsDir
  IfFileExists "$INSTDIR\owned-files.tsv" 0 no_manifest
  nsExec::ExecToLog '"$INSTDIR\packaging\uninstall_worker.exe" "$INSTDIR" "$PLUGINSDIR\pending.txt"'
  Pop $0
  ${If} $0 != 0
    MessageBox MB_ICONSTOP|MB_OK "Could not safely remove owned program files (code $0). Your files and uninstall registration were kept." /SD IDOK
    SetErrorLevel 1
    Abort
  ${EndIf}
  ClearErrors
  FileOpen $0 "$PLUGINSDIR\pending.txt" r
  IfErrors pending_error
pending_loop:
  ClearErrors
  FileReadUTF16LE $0 $1
  IfErrors pending_done
  StrCpy $1 $1 -1
  ClearErrors
  Delete "$1"
  IfErrors pending_delete_error
  Goto pending_loop
pending_done:
  FileClose $0
  Delete "$INSTDIR\owned-files.json"
  Delete "$INSTDIR\owned-files.tsv"
  Delete "$INSTDIR\packaging\uninstall_worker.exe"
  RMDir "$INSTDIR\packaging"
  Delete "$INSTDIR\Uninstall.exe"
  Delete "$DESKTOP\${PRODUCT}.lnk"
  Delete "$SMPROGRAMS\${PRODUCT}\${PRODUCT}.lnk"
  RMDir "$SMPROGRAMS\${PRODUCT}"
  DeleteRegKey HKCU "${REGKEY}"
  RMDir "$INSTDIR"
  Goto uninstall_done
pending_delete_error:
  FileClose $0
  MessageBox MB_ICONSTOP|MB_OK "Some program files are still in use. Close AES Studio and retry uninstall. Registration was kept." /SD IDOK
  SetErrorLevel 1
  Abort
pending_error:
  MessageBox MB_ICONSTOP|MB_OK "Could not read the owned-file cleanup list. Uninstall registration was kept." /SD IDOK
  SetErrorLevel 1
  Abort
no_manifest:
  MessageBox MB_ICONSTOP|MB_OK "The owned-file manifest is missing. No files were removed and the uninstall registration was kept." /SD IDOK
  SetErrorLevel 1
  Abort
uninstall_done:
SectionEnd






