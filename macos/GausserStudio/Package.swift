// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "GausserStudio",
    // MetalSplatter currently declares macOS 15 as its minimum supported platform.
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "GausserStudio", targets: ["GausserStudio"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/scier/MetalSplatter.git",
            revision: "2b965de1934de38dda1c71cf90bf798aa948a14c"
        ),
    ],
    targets: [
        .executableTarget(
            name: "GausserStudio",
            dependencies: [
                .product(name: "MetalSplatter", package: "MetalSplatter"),
                .product(name: "SplatIO", package: "MetalSplatter"),
            ]
        ),
        .testTarget(
            name: "GausserStudioTests",
            dependencies: [
                "GausserStudio",
                .product(name: "SplatIO", package: "MetalSplatter"),
            ]
        ),
    ]
)
