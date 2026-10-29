# Third-party notices for the AIWF Studio Windows preview

This bundle includes the AES Studio native application and the self-contained Windows App SDK runtime files produced by the locked native NuGet graph. The package licenses and notices listed below are copied into `licenses/nuget/`. Review each license before redistributing the bundle.

- `Microsoft.Web.WebView2` 1.0.4314.35: LICENSE.txt (file)
  - `licenses/nuget/Microsoft.Web.WebView2/LICENSE.txt`
  - `licenses/nuget/Microsoft.Web.WebView2/NOTICE.txt`
- `Microsoft.Windows.AI.MachineLearning` 2.1.74: license.txt (file)
  - `licenses/nuget/Microsoft.Windows.AI.MachineLearning/license.txt`
  - `licenses/nuget/Microsoft.Windows.AI.MachineLearning/ThirdPartyNotices.txt`
- `Microsoft.Windows.CppWinRT` 2.0.250303.1: LICENSE (file)
  - `licenses/nuget/Microsoft.Windows.CppWinRT/LICENSE`
- `Microsoft.Windows.ImplementationLibrary` 1.0.250325.1: LICENSE (file)
  - `licenses/nuget/Microsoft.Windows.ImplementationLibrary/LICENSE`
  - `licenses/nuget/Microsoft.Windows.ImplementationLibrary/ThirdPartyNotices.txt`
- `Microsoft.Windows.SDK.BuildTools` 10.0.26100.9169: https://aka.ms/WinSDKLicenseURL
- `Microsoft.Windows.SDK.BuildTools.MSIX` 1.7.260610101: preview_sdk_license.txt (file)
  - `licenses/nuget/Microsoft.Windows.SDK.BuildTools.MSIX/NOTICE.txt`
  - `licenses/nuget/Microsoft.Windows.SDK.BuildTools.MSIX/preview_sdk_license.txt`
- `Microsoft.Windows.SDK.CPP` 10.0.26100.9169: https://aka.ms/WinSDKLicenseURL
- `Microsoft.Windows.SDK.CPP.x64` 10.0.26100.9169: https://aka.ms/WinSDKLicenseURL
- `Microsoft.WindowsAppSDK` 2.5.1: license.txt (file)
  - `licenses/nuget/Microsoft.WindowsAppSDK/license.txt`
  - `licenses/nuget/Microsoft.WindowsAppSDK/NOTICE.txt`
- `Microsoft.WindowsAppSDK.AI` 2.5.5: license.txt (file)
  - `licenses/nuget/Microsoft.WindowsAppSDK.AI/license.txt`
- `Microsoft.WindowsAppSDK.Base` 2.0.4: license.txt (file)
  - `licenses/nuget/Microsoft.WindowsAppSDK.Base/license.txt`
  - `licenses/nuget/Microsoft.WindowsAppSDK.Base/NOTICE.txt`
- `Microsoft.WindowsAppSDK.DWrite` 2.1.0: license.txt (file)
  - `licenses/nuget/Microsoft.WindowsAppSDK.DWrite/license.txt`
- `Microsoft.WindowsAppSDK.Foundation` 2.3.12: license.txt (file)
  - `licenses/nuget/Microsoft.WindowsAppSDK.Foundation/license.txt`
- `Microsoft.WindowsAppSDK.InteractiveExperiences` 2.1.9: license.txt (file)
  - `licenses/nuget/Microsoft.WindowsAppSDK.InteractiveExperiences/license.txt`
- `Microsoft.WindowsAppSDK.ML` 2.1.94: license.txt (file)
  - `licenses/nuget/Microsoft.WindowsAppSDK.ML/license.txt`
- `Microsoft.WindowsAppSDK.Runtime` 2.5.1: license.txt (file)
  - `licenses/nuget/Microsoft.WindowsAppSDK.Runtime/license.txt`
  - `licenses/nuget/Microsoft.WindowsAppSDK.Runtime/NOTICE.txt`
- `Microsoft.WindowsAppSDK.Search` 2.5.5: license.txt (file)
  - `licenses/nuget/Microsoft.WindowsAppSDK.Search/license.txt`
- `Microsoft.WindowsAppSDK.Widgets` 2.0.5: license.txt (file)
  - `licenses/nuget/Microsoft.WindowsAppSDK.Widgets/license.txt`
- `Microsoft.WindowsAppSDK.WinUI` 2.3.9: license.txt (file)
  - `licenses/nuget/Microsoft.WindowsAppSDK.WinUI/license.txt`
  - `licenses/nuget/Microsoft.WindowsAppSDK.WinUI/NOTICE.txt`

The stable-diffusion.cpp CUDA 13 executable and its DLLs are included. Their source, license, and CUDA notices are in dependencies/stable-diffusion.cpp-cuda13/LICENSES/.

Microsoft Visual C++ 14.44 x64 redistributable DLLs are included app-locally; the redistribution notice is licenses/Microsoft-Visual-Cpp-Redist.txt. App-local copies require application updates to receive runtime security fixes.

Windows App SDK files are distributed under the included Microsoft license terms. The Microsoft documentation defines files binplaced by its NuGet package as redistributable code: https://learn.microsoft.com/windows/apps/package-and-deploy/self-contained-deploy/deploy-self-contained-apps


The bundled Python 3.12.13 runtime is the official python-build-standalone 20260310 x86-64 Windows install-only distribution. Its accompanying full-distribution license texts and metadata are in licenses/python-build-standalone/. Its pip and bundled dependencies include additional license texts under runtime/python/Lib/site-packages/. Each offline wheel includes its own distribution metadata and licenses.

The installer uses NSIS 3.13 with zlib compression; its notices are in licenses/NSIS-3.13-COPYING.txt.
