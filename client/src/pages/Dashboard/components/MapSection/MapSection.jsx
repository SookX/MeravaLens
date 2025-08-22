import { Box, Typography } from "@mui/material"
import Map from "./components/Map/Map"
import { mapHeight, Section, StyledContainer, StyledDivider } from "./styling"

const MapSection = ({ error = null }) => {


    return (
        <Section>
            <Box mb={6}>
                <Box mb={6} sx={{ textAlign: "center" }}>
                    <StyledDivider><Typography color="primary" variant="h2">Select a point</Typography></StyledDivider>
                    <Typography variant="body1">Click anywhere on the map and get a detailed analysis - segmented satellite picture as well as the latest weather and air pollution details.</Typography>
                    {
                        error &&
                        <Typography mt={1} variant="body1" color="error">{error}</Typography>
                    }
                </Box>

                <StyledContainer>
                    <Map mapHeight={mapHeight} />
                </StyledContainer>
            </Box>
        </Section>
    )
}

export default MapSection