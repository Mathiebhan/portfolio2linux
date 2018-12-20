#!/bin/bash
#scripts 

dd if=/dev/random bs=4 count=2 status=none | od -A none -t u4
