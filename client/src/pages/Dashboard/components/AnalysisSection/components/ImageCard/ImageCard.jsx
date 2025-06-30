import { Box, Card, Stack, styled, Typography } from "@mui/material"
import { theme } from "../../../../../../theme/theme"
import { useContext } from "react"
import { DashboardContext } from "../../../../Dashboard"


const ImageCard = () => {
    // Gets dashboard data
    const { image, segmentedImage } = useContext(DashboardContext)



    const StyledCard = styled(Card)(({ theme })=>({
        padding: `${theme.spacing(4)} ${theme.spacing(4)}`
    }))



    const ImageContainer = styled(Box)(({ theme })=>({
        borderRadius: theme.shape.borderRadius,
        overflow: "hidden"
    }))



    return (
        <StyledCard>
            <Box sx={{ width: "80%" }} mb={4}>
                <Typography variant="h3" color="primary">Satellite image</Typography>
                <Typography variant="body1">Bellow you can see a recent satellite shot of the selected place.</Typography>
                <Typography variant="body1">The right image is a segmentation of the image done by our AI model, showing what each part of the place is.</Typography>
            </Box>
            
            <Stack direction="row" gap={3} sx={{ position: "relative" }}>
                <ImageContainer flex={1}><img className="sat-img" src={image} /></ImageContainer>
                <ImageContainer flex={1}><img className="sat-segm-img" src={segmentedImage} /></ImageContainer>
            </Stack>
        </StyledCard>
    )
}

export default ImageCard