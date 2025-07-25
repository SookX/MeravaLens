import { GoogleMap, LoadScript, Marker } from "@react-google-maps/api";
import { useContext } from "react";
import { DashboardContext } from "../../../../Dashboard";
import { theme } from "../../../../../../theme/theme";

const Map = ({ mapHeight = 100 }) => {
    // Gets dashboard data
    const { coords, setCoords } = useContext(DashboardContext)



    const key = import.meta.env.VITE_GOOGLE_API_KEY

    // Stores map settings
    const mapStyles = {
        minHeight: `${mapHeight}px`,
        width: "100%",
        position: "absolute",
        top: 0,
        left: 0,
        borderRadius: theme.shape.borderRadius
    };

    const defaultCenter = {
        lat: 42.698334, // Latitude of default location (Sofia)
        lng: 23.319941 // Longitude of default location
    }

    const handleSelectCoords = (e) => {
        setCoords({ lat: e.latLng.lat(), lng: e.latLng.lng() })
    }

    return (
        <>
            <LoadScript googleMapsApiKey={key}>
                <GoogleMap
                    mapContainerStyle={mapStyles}
                    zoom={15}
                    center={coords ? coords : defaultCenter}
                    onClick={(e) => handleSelectCoords(e)}
                >
                    {
                        coords &&
                        <Marker position={coords} />
                    }
                </GoogleMap>
            </LoadScript>
        </>
    )
}

export default Map