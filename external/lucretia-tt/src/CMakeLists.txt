target_sources(app
    PRIVATE
        main.cpp
)

add_subdirectory(particles)
add_subdirectory(tracking)
add_subdirectory(io)
add_subdirectory(elements)
add_subdirectory(spacecharge)
