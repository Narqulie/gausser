// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "GausserStudio",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "GausserStudio", targets: ["GausserStudio"]),
    ],
    targets: [
        .executableTarget(name: "GausserStudio"),
        .testTarget(name: "GausserStudioTests", dependencies: ["GausserStudio"]),
    ]
)
