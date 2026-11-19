#include "targets/sensenova_loop_9b/impl/config.h"

#include <cstdio>

namespace {

using Config = ninfer::targets::sensenova_loop_9b::detail::TextConfig;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::fprintf(stderr, "%s\n", message);
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    failures += check(Config::physical_layers == 32, "physical layer count");
    failures += check(Config::execution_slots == 56, "execution slot count");
    failures += check(Config::full_attention_layers() == 8, "physical full layer count");
    failures += check(Config::gdn_layers() == 24, "physical GDN layer count");
    failures += check(Config::full_attention_slots == 14, "full state slot count");
    failures += check(Config::gdn_slots == 42, "GDN state slot count");

    failures += check(Config::slot_physical_layer(0) == 0, "slot 0 physical");
    failures += check(Config::slot_physical_layer(3) == 3, "slot 3 physical");
    failures += check(Config::slot_physical_layer(4) == 4, "slot 4 physical");
    failures += check(Config::slot_physical_layer(27) == 27, "slot 27 physical");
    failures += check(Config::slot_physical_layer(28) == 4, "slot 28 second visit");
    failures += check(Config::slot_physical_layer(51) == 27, "slot 51 second visit end");
    failures += check(Config::slot_physical_layer(52) == 28, "slot 52 tail");
    failures += check(Config::slot_physical_layer(55) == 31, "slot 55 physical");
    failures += check(Config::slot_is_full_attention(3), "slot 3 full attention");
    failures += check(!Config::slot_is_full_attention(4), "slot 4 GDN");
    failures += check(Config::slot_is_full_attention(7), "slot 7 full attention");
    failures += check(Config::slot_is_full_attention(31), "slot 31 physical 7 full attention");
    failures += check(Config::slot_is_full_attention(55), "slot 55 physical 31 full attention");

    failures += check(Config::physical_full_index(3) == 0, "physical 3 full index");
    failures += check(Config::physical_full_index(7) == 1, "physical 7 full index");
    failures += check(Config::physical_full_index(31) == 7, "physical 31 full index");
    failures += check(Config::physical_gdn_index(0) == 0, "physical 0 GDN index");
    failures += check(Config::physical_gdn_index(4) == 3, "physical 4 GDN index");
    failures += check(Config::physical_gdn_index(30) == 23, "physical 30 GDN index");

    failures += check(Config::slot_full_index(3) == 0, "slot 3 full state index");
    failures += check(Config::slot_full_index(7) == 1, "slot 7 full state index");
    failures += check(Config::slot_full_index(31) == 7, "slot 31 second-visit full state index");
    failures += check(Config::slot_gdn_index(4) == 3, "slot 4 GDN state index");
    failures += check(Config::slot_gdn_index(28) == 21, "slot 28 second-visit GDN state index");
    failures += check(Config::slot_gdn_index(4) != Config::slot_gdn_index(28),
                      "loop visits isolate GDN state");
    failures += check(Config::slot_full_index(7) != Config::slot_full_index(35),
                      "loop visits isolate full-attention KV");

    std::printf(failures == 0 ? "execution plan tests passed\n" : "execution plan tests failed\n");
    return failures == 0 ? 0 : 1;
}
