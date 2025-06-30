import { createContext, useContext, useEffect, useState } from "react"
import { DataContext } from "../../context/DataContext"
import temp from "../../img/temp.png"
import MapSection from "./components/MapSection/MapSection"
import AnalysisSection from "./components/AnalysisSection/AnalysisSection"

export const DashboardContext = createContext({  })

const Dashboard = () => {
    const { crud } = useContext(DataContext)


    // Holds the selected coordinates
    const [coords, setCoords] = useState(null)
    const [analysis, setAnalysis] = useState(false)
    const [image, setImage] = useState(null)
    const [segmentedImage, setSegmentedImage] = useState(temp)
    const [weather, setWeather] = useState(null)
    const [airPollution, setAirPollution] = useState(null)
    
    
    
    useEffect(() => {
        const fetching = async () => {
            const response = await crud({
                url: `/environment/?lat=${coords.lat}&lon=${coords.lng}`,
                method: "get"
            })

            console.log(response)
            if(response.status == 200) {
                setAnalysis(true)
                setImage(response.data.map_image_url),
                setWeather(response.data.weather)
                setAirPollution(response.data.air_pollution)
            }
        }

        if(coords) fetching()
    }, [coords])



    return (
        <DashboardContext.Provider
            value={{
                coords, setCoords,
                weather,
                airPollution,
                image,
                segmentedImage
            }}
        >
            <MapSection />

            {
                analysis &&
                <AnalysisSection />
            }
        </DashboardContext.Provider>
    )
}

export default Dashboard