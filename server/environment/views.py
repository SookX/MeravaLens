import requests
from django.conf import settings
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework import status
from rest_framework.permissions import IsAuthenticated;

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def environmental_data(request):
    api_key = settings.OPENWEATHER_API_KEY
    if not api_key:
        return Response({'error': 'OpenWeather API key not configured'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    api_key_google_maps = settings.GOOGLE_API_KEY
    if not api_key_google_maps:
        return Response({'error': 'Google Maps API key not configured'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    fast_api_url = settings.FAST_API_MICROSERVICE_URL
    if not fast_api_url:
        return Response({'error': 'FastAPI microservice URL not configured'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    lat = request.GET.get('lat')
    lon = request.GET.get('lon')

    if not lat or not lon:
        return Response({'error': 'Missing lat or lon query parameters'}, status=status.HTTP_400_BAD_REQUEST)

    timeout = 5
    location_params = {'lat': lat, 'lon': lon, 'appid': api_key}

    try:
        session = requests.Session()

        air_pollution_resp = session.get(
            'http://api.openweathermap.org/data/2.5/air_pollution',
            params=location_params,
            timeout=timeout
        )
        air_pollution_resp.raise_for_status()

        weather_resp = session.get(
            'https://api.openweathermap.org/data/2.5/weather',
            params=location_params,
            timeout=timeout
        )
        weather_resp.raise_for_status()

        zoom = 18
        size = "1024x1024"
        maptype = "satellite"
        map_url = (
            f"https://maps.googleapis.com/maps/api/staticmap"
            f"?center={lat},{lon}"
            f"&zoom={zoom}"
            f"&size={size}"
            f"&maptype={maptype}"
            f"&key={api_key_google_maps}"
        )

        post_payload = {
            "url": map_url  
        }

        fastapi_response = requests.post(
            fast_api_url,
            json=post_payload,
            # timeout=timeout
        )
        fastapi_response.raise_for_status()

        return Response({
            'location': {
                'latitude': lat,
                'longitude': lon
            },
            'air_pollution': air_pollution_resp.json(),
            'weather': weather_resp.json(),
            'map_image_url': map_url,
            'fastapi_result': fastapi_response.json()
        })

    except requests.exceptions.RequestException as e:
        return Response({'error': str(e)}, status=status.HTTP_502_BAD_GATEWAY)
