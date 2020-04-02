# Soft Nearest Neighbor Loss
# Copyright (C) 2020  Abien Fred Agarap
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Sample module for using Autoencoder with SNNL"""

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


def parse_args():
    parser = argparse.ArgumentParser(description="Autoencoder with SNNL")
    group = parser.add_argument_group("Parameters")
    group.add_argument(
        "-s",
        "--seed",
        required=False,
        default=1234,
        type=int,
        help="the random seed value to use, default: [1234]",
    )
    group.add_argument(
        "-d",
        "--device",
        required=False,
        default="cpu",
        type=str,
        help="the device to use, default: [cpu]",
    )
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    pass
