#!/bin/bash

id="$(gdrive list | grep PixelDrawEnv.py | awk '{print $1}')"

"$(gdrive delete $id)"
"$( gdrive upload PixelDrawEnv.py -p 1BWFHMrcgZeBYgM3kEp1jCB-2_6NcljT-)"