#include "utils.h"

#include <QDebug>
#include <QFile>

void validate(bool suc, QString msg) {
    if(!suc) {
        qDebug() << "ERROR: " << msg;
        exit(-1);
    }
}

bool readShader(QString fileName, std::vector<uint32_t> &code) {
    QFile fi(fileName);
    if(!fi.open(QIODevice::ReadOnly)) {
        qDebug() << "ERROR: could not read shader file:" << fileName;
        return false;
    }
    qint64 sizeInBytes = fi.size();
    if(sizeInBytes > INT_MAX) {
        qDebug() << "shader size large. channge read code: " << fileName;
        fi.close();
        return false;
    }
    if(sizeInBytes % sizeof(uint32_t) != 0) {
        qDebug() << "shader size not divisible as set of uint32_t's: " << fileName;
        fi.close();
        return false;
    }
    QDataStream ds(&fi);
    int64_t  size = sizeInBytes / sizeof(uint32_t);
    code.resize(size);
    int len = ds.readRawData((char *)code.data(),sizeInBytes);
    if(len != sizeInBytes) {
        qDebug() << "ERROR: all data not read from shader file:" << fileName;
        fi.close();
        return false;
    }
    fi.close();
    return true;
}
