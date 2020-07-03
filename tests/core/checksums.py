#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt
import hashlib
import zlib

test = b"Test this string"

sha256_comp = "%x" % gpt.sha256(test)

m = hashlib.sha256()
m.update(test)
sha256_ref = m.hexdigest()

gpt.message(sha256_comp, sha256_ref)

assert sha256_comp == sha256_ref

crc32_comp = "%x" % gpt.crc32(test)
crc32_ref = "%x" % zlib.crc32(test)

gpt.message(crc32_comp, crc32_ref)
assert crc32_comp == crc32_ref

gpt.message("Tests successful")
