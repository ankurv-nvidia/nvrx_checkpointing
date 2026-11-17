set(VCPKG_BUILD_TYPE release)

set(release 20261110)
set(archive_name "cpython-${VERSION}+${release}-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz")
string(REPLACE "+" "%2B" archive_url_name "${archive_name}")
vcpkg_download_distfile(archive
    URLS
        "https://github.com/astral-sh/python-build-standalone/releases/download/${release}/${archive_url_name}"
        "https://releases.astral.sh/github/python-build-standalone/releases/download/${release}/${archive_url_name}"
    FILENAME "${archive_name}"
    SHA512 ed8c29f5fcd4adec35942af15a97cbb57c70a39ce53a74dbcfe44f039dfb5c0eacf41caa78c3ccff124a829bcdfd09eecb7130c72c48de1d7d2f230aebcbab57
)
vcpkg_extract_source_archive(source_path ARCHIVE "${archive}")

string(REGEX MATCH "^[0-9]+\\.[0-9]+" minor_version "${VERSION}")
file(INSTALL "${source_path}/include/python${minor_version}" DESTINATION "${CURRENT_PACKAGES_DIR}/include")
file(INSTALL
    "${source_path}/lib/libpython${minor_version}.so"
    "${source_path}/lib/libpython${minor_version}.so.1.0"
    DESTINATION "${CURRENT_PACKAGES_DIR}/lib"
)
vcpkg_install_copyright(FILE_LIST "${source_path}/lib/python${minor_version}/LICENSE.txt")
