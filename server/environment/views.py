import requests
from django.conf import settings
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status

@api_view(['GET'])
def environmental_data(request):
    api_key = settings.OPENWEATHER_API_KEY

    lat = request.GET.get('lat')
    lon = request.GET.get('lon')

    if not lat or not lon:
        return Response({'error': 'Missing lat or lon query parameters'}, status=status.HTTP_400_BAD_REQUEST)

    session = requests.Session()
    timeout = 5

    location_params = {'lat': lat, 'lon': lon, 'appid': api_key}
    appid_only = {'appid': api_key}

    try:
        air_pollution_resp = session.get(
            'http://api.openweathermap.org/data/2.5/air_pollution',
            params=location_params,
            timeout=timeout
        )
        weather_resp = session.get(
            'https://api.openweathermap.org/data/2.5/weather',
            params=location_params,
            timeout=timeout
        )
        # road_risk_resp = session.get(
        #     'https://api.openweathermap.org/data/2.5/roadrisk',
        #     params=appid_only,
        #     timeout=timeout
        # )

        air_pollution_resp.raise_for_status()
        weather_resp.raise_for_status()
        # road_risk_resp.raise_for_status()

        return Response({
            'location': {
                'latitude': lat,
                'longitude': lon
            },
            'air_pollution': air_pollution_resp.json(),
            'weather': weather_resp.json(),
            # 'road_risk': road_risk_resp.json()
        })

    except requests.exceptions.RequestException as e:
        return Response({'error': str(e)}, status=status.HTTP_502_BAD_GATEWAY)
