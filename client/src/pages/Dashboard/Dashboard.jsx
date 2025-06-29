import { useContext, useEffect, useState } from "react"
import Map from "./components/Map"
import { DataContext } from "../../context/DataContext"
import { Box, Divider, styled, Typography } from "@mui/material"
import { theme } from "../../theme/theme"
import temp from "../../img/temp.png"

const Dashboard = () => {
    const { crud } = useContext(DataContext)


    // Holds the selected coordinates
    const [coords, setCoords] = useState(null)
    const [analysis, setAnalysis] = useState(false)
    const [image, setImage] = useState(null)
    const [segmentedImage, setSegmentedImage] = useState(temp)
    
    
    
    useEffect(() => {
        const fetching = async () => {
            const response = await crud({
                url: `/environment/?lat=${coords.lat}&lon=${coords.lng}`,
                method: "get"
            })

            console.log(response)
            if(response.status == 200) {
                setAnalysis(true)
                setImage(response.data.map_image_url)
            }
        }

        if(coords) fetching()
    }, [coords])



    const StyledContainer = styled(Box)(({ theme })=>({
        borderRadius: theme.shape.borderRadius,
        overflow: "hidden"
    }))



    const MapSection = styled(Box)(({ theme })=>({
        width: "50%",
        margin: "0 auto"
    }))



    const StyledDivider = styled(Divider)(({ theme })=>({
        "&::before, &::after": {
            borderColor: theme.palette.text.default
        }
    }))



    return (
        <>
            <MapSection>
                <Box mb={6}>
                    <Box mb={4}>
                        <StyledDivider><Typography textAlign={"center"} color="primary" variant="h2">Select a point</Typography></StyledDivider>
                        <Typography textAlign="center" variant="body1">Click anywhere on the map and get a detailed analysis - segmented satellite picture as well as the latest weather and air pollution details.</Typography>
                    </Box>

                    <StyledContainer>
                        <Map coords={coords} setCoords={setCoords} />
                    </StyledContainer>
                </Box>
            </MapSection>

            {
                analysis &&
                <Box sx={{ width: "50%", margin: "0 auto" }}>
                    <Box sx={{ position: "relative" }}>
                        <img className="sat-img" src={image} />
                        <img className="sat-segm-img" src={segmentedImage} />
                    </Box>
                </Box>
            }
        </>
    )
}

export default Dashboard