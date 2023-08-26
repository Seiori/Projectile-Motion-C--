# Automatically generated by boost-vcpkg-helpers/generate-ports.ps1

include(vcpkg_common_functions)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO boostorg/gil
    REF boost-1.68.0
    SHA512 d3a965aca410f91c214d8f433273e08eefcc20f0e470baa2aba3385ce45a0e18307aae3b81ea619abe39066e4fd924476b7f29c8f45e1aff25a2a7d3eb4062fb
    HEAD_REF master
)

include(${CURRENT_INSTALLED_DIR}/share/boost-vcpkg-helpers/boost-modular-headers.cmake)
boost_modular_headers(SOURCE_PATH ${SOURCE_PATH})
