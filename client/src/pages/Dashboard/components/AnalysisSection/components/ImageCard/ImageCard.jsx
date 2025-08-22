import { Box, Stack, Typography } from "@mui/material"
import { useContext } from "react"
import { DashboardContext } from "../../../../Dashboard"
import { ImageContainer, StyledCard } from "./styling"


const ImageCard = () => {
    // Gets dashboard data
    const { image, segmentedImage } = useContext(DashboardContext)



    return (
        <StyledCard>
            <Box sx={{ width: "80%" }} mb={4}>
                <Typography variant="h3" color="primary">Satellite image</Typography>
                <Typography variant="body1">Bellow you can see a recent satellite shot of the selected place.</Typography>
                <Typography variant="body1">The second image is a segmentation of the image done by our AI model, showing what each part of the place is.</Typography>
            </Box>
            
            <Stack direction={{ xs: "column", md: "row"}} gap={3} sx={{ position: "relative" }}>
                <ImageContainer flex={1}><img className="sat-img" src={image} /></ImageContainer>
                <ImageContainer flex={1}><img className="sat-segm-img" src={segmentedImage} /></ImageContainer>
            </Stack>
        </StyledCard>
    )
}

export default ImageCard