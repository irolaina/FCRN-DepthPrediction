#!/usr/bin/env bash
find -name 'city_*' -exec bash -c 'mv $0 ${0/city_}' {} \;
find -name 'residential_*' -exec bash -c 'mv $0 ${0/residential_}' {} \;
find -name 'campus_*' -exec bash -c 'mv $0 ${0/campus_}' {} \;
