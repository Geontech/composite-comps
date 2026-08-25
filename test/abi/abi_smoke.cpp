// End-to-end smoke for the single create() ABI (#13). For each component .so: dlopen it,
// verify the composite_abi_version() handshake, call the single create(id, args) signature
// with a CUSTOM id (not the component's old hardcoded name), and assert the returned
// component honors that id. This directly exercises the fixes: file_writer/histogram were
// loadable only under their hardcoded id, and exp_smooth was unloadable through the loader
// at all (its create(type) misread the id as the type). Also checks a bad type fails cleanly.
#include <composite/core/register.hpp>

#include <dlfcn.h>

#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <string_view>

using create_fn = std::shared_ptr<composite::component> (*)(std::string_view, const composite::create_args&);
using abi_fn = unsigned long (*)();

static int failures = 0;

static void check_load(const std::string& lib, const char* id, const char* type, bool expect_ok) {
    void* h = dlopen(lib.c_str(), RTLD_NOW);
    if (h == nullptr) { std::printf("FAIL dlopen %s: %s\n", lib.c_str(), dlerror()); ++failures; return; }

    auto abi = reinterpret_cast<abi_fn>(dlsym(h, "composite_abi_version"));
    if (abi == nullptr) { std::printf("FAIL %s: no composite_abi_version symbol\n", lib.c_str()); ++failures; dlclose(h); return; }
    if (abi() != composite::abi_version) {
        std::printf("FAIL %s: abi %lu != framework %lu\n", lib.c_str(), abi(), composite::abi_version); ++failures; dlclose(h); return;
    }
    auto create = reinterpret_cast<create_fn>(dlsym(h, "create"));
    if (create == nullptr) { std::printf("FAIL %s: no create symbol\n", lib.c_str()); ++failures; dlclose(h); return; }

    composite::create_args args;
    if (type != nullptr) { args.values = composite::properties::json{{"type", type}}; }

    std::shared_ptr<composite::component> c;
    bool threw = false;
    try { c = create(id, args); } catch (const std::exception&) { threw = true; }

    if (expect_ok) {
        if (!c) { std::printf("FAIL %s: create returned null (id=%s type=%s)\n", lib.c_str(), id, type ? type : "-"); ++failures; }
        else if (c->id() != id) { std::printf("FAIL %s: id '%s' != expected '%s' (id NOT honored)\n", lib.c_str(), c->id().c_str(), id); ++failures; }
        else { std::printf("OK   %-14s id=%-6s type=%s\n", c->id().c_str(), id, type ? type : "-"); }
    } else {
        if (c && !threw) { std::printf("FAIL %s: expected rejection but loaded (id=%s type=%s)\n", lib.c_str(), id, type ? type : "-"); ++failures; }
        else { std::printf("OK   rejected bad type (lib=%s type=%s)\n", lib.c_str(), type ? type : "-"); }
    }
    c.reset();   // ~component before dlclose (its vtable lives in the mapping)
    dlclose(h);
}

int main(int argc, char** argv) {
    // Driven by argv so test/abi/CMakeLists.txt can generate one ctest per ENABLED module
    // rather than carrying a hardcoded list that silently misses the next component added.
    //
    // usage: abi_smoke <module.so> <id> [type] [--expect-reject]
    if (argc < 3) {
        std::printf("usage: abi_smoke <module.so> <id> [type] [--expect-reject]\n");
        return 2;
    }
    const std::string lib = argv[1];
    const char* id = argv[2];
    const char* type = nullptr;
    bool expect_ok = true;
    for (int i = 3; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--expect-reject") {
            expect_ok = false;
        } else {
            type = argv[i];
        }
    }

    check_load(lib, id, type, expect_ok);

    std::printf(failures ? "\n%d FAILURE(S)\n" : "\nABI SMOKE PASSED\n", failures);
    return failures ? 1 : 0;
}
