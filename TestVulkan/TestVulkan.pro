TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle

DEFINES += DEV_BUILD

SOURCES += \
        Headless.cpp \
        main.cpp \
        utils.cpp

HEADERS += \
    Headless.hpp \
    utils.h

RESOURCES += \
    shaders.qrc

unix:!macx{
    LIBS += -lvulkan
}

win32-msvc*{
    INCLUDEPATH += $$(VULKAN_SDK)/Include/
    LIBS += -L"$$(VULKAN_SDK)/Lib/" -lvulkan-1
}
