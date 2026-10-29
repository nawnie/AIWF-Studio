!ifndef AES_SYSTEM_REQUIREMENTS_NSH
!define AES_SYSTEM_REQUIREMENTS_NSH

!define AES_MIN_WINDOWS_BUILD 19041

; Keep the decision separate from the OS probes so each outcome can be tested.
!macro AES_RequirementStatus result is64 build
  StrCpy ${result} "ok"
  ${If} ${is64} != 1
    StrCpy ${result} "architecture"
  ${ElseIf} ${build} < ${AES_MIN_WINDOWS_BUILD}
    StrCpy ${result} "version"
  ${EndIf}
!macroend

!endif
