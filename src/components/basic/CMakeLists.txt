# Library
add_library(basic MODULE
    component.cpp
)
# Link
target_link_libraries(basic
    PRIVATE
    caddie::caddie
)
# Install
install(TARGETS basic
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
