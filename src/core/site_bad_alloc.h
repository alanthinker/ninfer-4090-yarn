#pragma once

#include <new>

namespace ninfer::core {

// std::bad_alloc carrying the reservation site that failed.  Runtime resource exhaustion is
// reported as bad_alloc, so without a site tag a generic handler (the serving request path)
// cannot tell which pool actually ran out.  The message is a static string literal.
class SiteBadAlloc final : public std::bad_alloc {
public:
    explicit SiteBadAlloc(const char* site) noexcept : site_(site) {}

    [[nodiscard]] const char* what() const noexcept override { return site_; }

private:
    const char* site_;
};

}  // namespace ninfer::core

#define NINFER_SITE_BAD_ALLOC(site) throw ::ninfer::core::SiteBadAlloc(site)
