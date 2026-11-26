// Source-compiled HIP device identity: no architecture spoofing or guessed struct ABI.
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>
int main(int argc, char** argv) {
  bool wsl = argc == 3 && std::string(argv[2]) == "--wsl";
  if (argc != 2 && !wsl) return 2;
  char* end = nullptr;
  long index = std::strtol(argv[1], &end, 10);
  if (!end || *end || index < 0 || index > 1024) return 2;
  hipDeviceProp_t properties{};
  if (hipGetDeviceProperties(&properties, static_cast<int>(index)) != hipSuccess) return 1;
  if (wsl) {
    int count = 0;
    if (index != 0 || hipGetDeviceCount(&count) != hipSuccess || count != 1) return 1;
    // Verify HIP actually opened the DXG backend and host shim for this device.
    // RTLD_NOLOAD checks residency without loading either library on our behalf.
    if (!dlopen("librocdxg.so.1", RTLD_NOW | RTLD_NOLOAD) ||
        !dlopen("libdxcore.so", RTLD_NOW | RTLD_NOLOAD)) return 1;
  }
  char bus[32]{};
  hipUUID uuid{};
  int driver = 0;
  if (hipDeviceGetPCIBusId(bus, sizeof(bus), index) != hipSuccess ||
      hipDeviceGetUuid(&uuid, index) != hipSuccess ||
      hipDriverGetVersion(&driver) != hipSuccess) return 1;
  std::cout << "{\"accelerator\":\"rocm\",\"gpu_index\":" << index
            << ",\"name\":" << std::quoted(properties.name)
            << ",\"pci_bus_id\":" << std::quoted(bus)
            << ",\"driver_version\":" << driver << ",\"uuid\":\"";
  for (unsigned char byte : uuid.bytes)
    std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
  std::string target(properties.gcnArchName);
  target = target.substr(0, target.find(':'));
  std::cout << "\",\"gfx_target\":" << std::quoted(target) << "}\n";
}
