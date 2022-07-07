#ifndef UTILS_H
#define UTILS_H

#include <QString>

void validate(bool suc, QString msg);
bool readShader(QString fileName, std::vector<uint32_t> &code);

#endif // UTILS_H
