import { createContext, useContext, useEffect, useState } from "react"
import { DataContext } from "../../context/DataContext"
import MapSection from "./components/MapSection/MapSection"
import AnalysisSection from "./components/AnalysisSection/AnalysisSection"

export const DashboardContext = createContext({  })

const Dashboard = () => {
    // Gets global data from the context
    const { crud, access, navigate } = useContext(DataContext)



    // Checks if the user is authenticated
    useEffect(() => {
        if(!access) navigate('/login')
    }, [access])



    // Holds the selected coordinates
    const [coords, setCoords] = useState(null)



    // Holds the analysis data
    const [analysis, setAnalysis] = useState(false)
    const [image, setImage] = useState(null)
    const [segmentedImage, setSegmentedImage] = useState(null)
    const [weather, setWeather] = useState(null)
    const [airPollution, setAirPollution] = useState(null)
    const [summary, setSummary] = useState(null)



    // Splits the summary into paragraphs
    const splitSummary = (raw) => {
        let components = raw.split("\n\n")
        
        components.splice(0, 1)
        let each = components[1]
        components = components.map(component => {
            return {
                title: component.split("**")[1],
                body: component.split("**")[2].split(": ")[1]
            }
        })

        let title = each.split("**")[1]
        let body = each.split("\n")
        body.splice(0, 1)

        body = body.map(component => {
            return {
                title: component.split("**")[1],
                body: component.split(": ")[1]
            }
        })

        each = {
            title,
            paragraphs: body
        }

        components[1] = each
        setSummary(components)
    }
    
    
    
    useEffect(() => {
        const fetching = async () => {
            const response = await crud({
                url: `/environment/?lat=${coords.lat}&lon=${coords.lng}`,
                method: "get"
            })

            console.log(response)
            if(response.status == 200) {
                setImage(response.data.map_image_url)
                setSegmentedImage(`data:image/jpeg;base64,${response.data.fastapi_result.segmentation_mask_base64}`)
                setWeather(response.data.weather)
                setAirPollution(response.data.air_pollution)
                splitSummary(response.data.fastapi_result.analysis)
                setAnalysis(true)
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
                segmentedImage,
                summary
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