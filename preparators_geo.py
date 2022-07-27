from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="DCEM") # maximum number of requests is 1 per second

def get_normalized_address(address):
    """
    Get normalized address
    """
    location = geolocator.geocode(address)
    return location.address

def get_address_json(address):
    """
    Get address json
    """
    location = geolocator.geocode(address)
    return location.raw


def get_coordinates(address):
    """
    Get coordinates from address
    """
    location = geolocator.geocode(address)
    return location.latitude, location.longitude



def main():
    address = '175 5th Avenue NYC'
    print(get_normalized_address(address))
    print(get_address_json(address))
    lat, lon = get_coordinates(address)
    print(lat, lon)


if __name__ == '__main__':
    main()