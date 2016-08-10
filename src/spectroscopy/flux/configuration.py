# Copyright (C) Nial Peters 2015
#
# This file is part of gns_flyspec.
#
# gns_flyspec is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gns_flyspec is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gns_flyspec.  If not, see <http://www.gnu.org/licenses/>.

import spectroscopy.flux
import json
import os

def load_config():
    path = os.path.join(spectroscopy.flux.__path__[0], "flyspec_config.cfg")

    with open(path, "rb") as ifp:
        config = json.load(ifp)

    return config
