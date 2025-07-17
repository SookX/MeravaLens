import { createContext, useContext, useEffect, useRef, useState } from "react"
import { DataContext } from "../../context/DataContext"
import MapSection from "./components/MapSection/MapSection"
import AnalysisSection from "./components/AnalysisSection/AnalysisSection"
import { crud } from "../../api/crud"
import { useNavigate } from "react-router-dom"
import { Box } from "@mui/material"

export const DashboardContext = createContext({  })

const Dashboard = () => {
    // Gets global data from the context
    const { access, setLoading } = useContext(DataContext)



    // Navigates users to another page
    const navigate = useNavigate()



    // Checks if the user is authenticated
    useEffect(() => {
        if(!access) navigate('/login')
    }, [access])



    // Holds the analysis section ref
    const analysisRef = useRef()



    // Holds the selected coordinates
    const [coords, setCoords] = useState(null)



    // Holds the analysis data
    const [analysis, setAnalysis] = useState(false)
    const [image, setImage] = useState(null)
    const [segmentedImage, setSegmentedImage] = useState(null)
    const [weather, setWeather] = useState(null)
    const [airPollution, setAirPollution] = useState(null)
    const [summary, setSummary] = useState(null)



    // Holds the error state
    const [error, setError] = useState(null)



    // Resets the loading state
    useEffect(() => {
        if(image && segmentedImage && weather && airPollution && summary) {
            setAnalysis(true)
            setLoading(false)
            setError(null)
        }
    }, [image, segmentedImage, weather, airPollution, summary])



    // Scrolls to the analysis
    useEffect(() => {
        if(analysis) analysisRef.current.scrollIntoView()
    }, [analysis])



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
            setLoading(true)
            setAnalysis(false)

            const response = await crud({
                url: `/environment/?lat=${coords.lat}&lon=${coords.lng}`,
                method: "get"
            })

            if(response.status == 200) {
                setImage(response.data.map_image_url)
                setSegmentedImage(`data:image/jpeg;base64,${response.data.fastapi_result.segmentation_mask_base64}`)
                setWeather(response.data.weather)
                setAirPollution(response.data.air_pollution)
                splitSummary(response.data.fastapi_result.analysis)
            }
            else {
                setError("Something went wrong with the analysis. Please try again.")
                setLoading(false)
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
            <MapSection error={error} />

            <Box ref={analysisRef}>
                {
                    analysis &&
                    <AnalysisSection />
                }
            </Box>
        </DashboardContext.Provider>
    )
}

export default Dashboard