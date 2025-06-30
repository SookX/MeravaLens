import { Box, Divider, styled, Typography } from "@mui/material"
import { theme } from "../../../../theme/theme"
import Map from "./components/Map/Map"

const MapSection = () => {
    const Section = styled(Box)(({ theme })=>({
        padding: `${theme.spacing(8)} ${theme.spacing(24)}`
    }))



    const StyledDivider = styled(Divider)(({ theme })=>({
        "&::before, &::after": {
            borderColor: theme.palette.text.default
        }
    }))



    const StyledContainer = styled(Box)(({ theme })=>({
        borderRadius: theme.shape.borderRadius,
        overflow: "hidden",
        width: "75%",
        margin: "0 auto"
    }))



    return (
        <Section>
            <Box mb={6}>
                <Box mb={6}>
                    <StyledDivider><Typography textAlign={"center"} color="primary" variant="h2">Select a point</Typography></StyledDivider>
                    <Typography textAlign="center" variant="body1">Click anywhere on the map and get a detailed analysis - segmented satellite picture as well as the latest weather and air pollution details.</Typography>
                </Box>

                <StyledContainer>
                    <Map />
                </StyledContainer>
            </Box>
        </Section>
    )
}

export default MapSection