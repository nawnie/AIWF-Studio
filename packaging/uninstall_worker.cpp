#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <bcrypt.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <cstdio>
#pragma comment(lib, "bcrypt.lib")
namespace fs = std::filesystem;

struct Entry { fs::path relative; std::string hash; };
std::wstring widen(std::string const& text) {
    if (text.empty()) return {};
    int n=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,text.data(),static_cast<int>(text.size()),nullptr,0);
    if (!n) throw 1;
    std::wstring out(n,L'\0');
    if (!MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,text.data(),static_cast<int>(text.size()),out.data(),n)) throw 1;
    return out;
}
bool plain_path(fs::path const& root, fs::path const& rel) {
    fs::path current=root;
    auto attr=GetFileAttributesW(current.c_str());
    if (attr==INVALID_FILE_ATTRIBUTES || (attr & FILE_ATTRIBUTE_REPARSE_POINT)) return false;
    for (auto const& part : rel) {
        current /= part;
        attr=GetFileAttributesW(current.c_str());
        if (attr==INVALID_FILE_ATTRIBUTES) return false;
        if (attr & FILE_ATTRIBUTE_REPARSE_POINT) return false;
    }
    return true;
}
std::string sha256(fs::path const& p) {
    std::ifstream in(p,std::ios::binary);
    if (!in) return {};
    BCRYPT_ALG_HANDLE alg=nullptr; BCRYPT_HASH_HANDLE hash=nullptr;
    DWORD objlen=0,cb=0; std::vector<unsigned char> digest(32);
    if (BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0) return {};
    if (BCryptGetProperty(alg,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&objlen),sizeof(objlen),&cb,0)<0) {BCryptCloseAlgorithmProvider(alg,0);return {};}
    std::vector<unsigned char> object(objlen),buffer(1024*1024);
    if (BCryptCreateHash(alg,&hash,object.data(),objlen,nullptr,0,0)<0) {BCryptCloseAlgorithmProvider(alg,0);return {};}
    bool ok=true;
    while (in) {in.read(reinterpret_cast<char*>(buffer.data()),buffer.size()); auto n=in.gcount(); if(n && BCryptHashData(hash,buffer.data(),static_cast<ULONG>(n),0)<0){ok=false;break;}}
    if (!in.eof()) ok=false;
    if (ok && BCryptFinishHash(hash,digest.data(),static_cast<ULONG>(digest.size()),0)<0) ok=false;
    BCryptDestroyHash(hash); BCryptCloseAlgorithmProvider(alg,0);
    if (!ok) return {};
    static char const digits[]="0123456789abcdef"; std::string out; out.reserve(64);
    for(auto c:digest){out.push_back(digits[c>>4]);out.push_back(digits[c&15]);}
    return out;
}
int wmain(int argc,wchar_t** argv) {
    if(argc!=3) return 2;
    try {
        fs::path root=fs::path(argv[1]).lexically_normal();
        if(!root.is_absolute() || root==root.root_path() || !plain_path(root,{})) return 3;
        if (!plain_path(root, fs::path(L"owned-files.tsv"))) return 4;
        std::ifstream manifest(root/L"owned-files.tsv",std::ios::binary);
        if(!manifest) return 4;
        std::string line;
        if(!std::getline(manifest,line) || line!="AES-STUDIO-OWNED-V1") return 5;
        std::vector<Entry> entries;
        while(std::getline(manifest,line)) {
            if(!line.empty() && line.back()=='\r') line.pop_back();
            auto tab=line.find('\t');
            if(tab==std::string::npos || tab!=64 || line.find('\t',tab+1)!=std::string::npos) return 6;
            auto hash=line.substr(0,tab);
            if(!std::all_of(hash.begin(),hash.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');})) return 6;
            auto raw=widen(line.substr(tab+1)); fs::path rel(raw);
            if(raw.empty() || rel.is_absolute() || rel.has_root_path() || raw.find(L':')!=std::wstring::npos || raw.find_first_of(L"*?\"<>|")!=std::wstring::npos) return 6;
            for(auto const& part:rel) if(part==L"." || part==L".." || part.empty()) return 6;
            entries.push_back({rel,hash});
        }
        if(!manifest.eof()) return 6;
        HANDLE pending=CreateFileW(argv[2],GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
        if(pending==INVALID_HANDLE_VALUE) return 7;
        bool outputOk=true; std::set<fs::path> directories;
        for(auto const& e:entries) {
            fs::path path=root/e.relative;
            for(auto parent=path.parent_path();parent!=root && !parent.empty();parent=parent.parent_path()) directories.insert(parent);
            if (!plain_path(root,e.relative) || sha256(path)!=e.hash) continue;
            if (!DeleteFileW(path.c_str())) {
                auto pendingLine=path.wstring()+L"\n"; DWORD written=0; DWORD bytes=static_cast<DWORD>(pendingLine.size()*sizeof(wchar_t));
                if(!WriteFile(pending,pendingLine.data(),bytes,&written,nullptr) || written!=bytes){outputOk=false;break;}
            }
        }
        CloseHandle(pending);
        // These two directories are created empty by the bundled Python runtime
        // and venv bootstrap, so they have no file entries in the manifest.
        // RemoveDirectoryW only removes an empty directory; never follow or
        // remove a user-supplied reparse point. Nonempty directories stay.
        for (fs::path const& rel : {fs::path(L"runtime\\python\\libs"), fs::path(L"venv\\Include")}) {
            if (plain_path(root, rel)) RemoveDirectoryW((root / rel).c_str());
        }
        std::vector<fs::path> sorted(directories.begin(),directories.end());
        std::sort(sorted.begin(),sorted.end(),[](auto const& a,auto const& b){return a.native().size()>b.native().size();});
        for(auto const& dir:sorted) {
            auto rel=dir.lexically_relative(root);
            // Manifest parent directories may have been replaced by a user junction.
            // Keep the junction itself as well as everything behind it.
            if (!rel.empty() && plain_path(root,rel)) RemoveDirectoryW(dir.c_str());
        }
        return outputOk?0:8;
    } catch(...) {return 9;}
}
