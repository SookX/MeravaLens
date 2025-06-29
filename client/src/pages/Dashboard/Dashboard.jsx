import { useContext, useEffect, useState } from "react"
import { DataContext } from "../../context/DataContext"
import { Typography } from "@mui/material"
import Map from "./components/Map/Map"
import { GoogleMap, LoadScript, Marker } from "@react-google-maps/api";

const Dashboard = () => {
    const { crud } = useContext(DataContext)


    // Holds the selected coordinates
    const [coords, setCoords] = useState(null)
    
    
    
    useEffect(() => {
        const fetching = async () => {
            const response = await crud({
                url: `/environment/?lat=${coords.lat}&lon=${coords.lng}`,
                method: "get"
            })

            console.log(response)
        }

        if(coords) {
            console.log(coords)
            fetching()
        }
    }, [coords])


    const key = import.meta.env.VITE_GOOGLE_API_KEY

    // Stores map settings
    const mapStyles = {
        height: "100vh",
        width: "100%",
        osition: "absolute",
        top: 0,
        left: 0

    };
    const defaultCenter = {
        lat: 42.698334, // Latitude of default location (Sofia)
        lng: 23.319941 // Longitude of default location
    };

    const handleSelectCoords = (e) => {
        setCoords({ lat: e.latLng.lat(), lng: e.latLng.lng() })
    }

    return (
        <>
            <LoadScript googleMapsApiKey={key}>
                <GoogleMap
                    mapContainerStyle={mapStyles}
                    zoom={15}
                    center={defaultCenter}
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

export default Dashboard