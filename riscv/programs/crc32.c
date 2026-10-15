/* Bytes, shifts, and XOR: bitwise CRC-32 of a string. */
static const char message[] = "The quick brown fox jumps over the lazy dog";

int main(void) {
    unsigned crc = 0xFFFFFFFFu;
    for (const char *p = message; *p; p++) {
        crc ^= (unsigned char)*p;
        for (int k = 0; k < 8; k++) crc = (crc >> 1) ^ (0xEDB88320u & -(crc & 1));
    }
    return ~crc == 0x414FA339u ? 0 : 1;
}
