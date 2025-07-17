import { Box, Divider, styled, Typography } from "@mui/material"
import { theme } from "../../../../theme/theme"
import Map from "./components/Map/Map"

const MapSection = ({ error = null }) => {
    const mapHeight = 500



    const Section = styled(Box)(({ theme })=>({
        padding: `${theme.spacing(8)} ${theme.spacing(24)}`,
        [theme.breakpoints.down("lg")]: { padding: `${theme.spacing(8)} ${theme.spacing(8)}` },
        [theme.breakpoints.down("sm")]: { padding: `${theme.spacing(8)} ${theme.spacing(2)}` }
    }))



    const StyledDivider = styled(Divider)(({ theme })=>({
        "&::before, &::after": {
            borderColor: theme.palette.text.default
        }
    }))



    const StyledContainer = styled(Box)(({ theme })=>({
        width: "75%",
        [theme.breakpoints.down("md")]: { width: "100%" },
        margin: "0 auto",
        position: "relative",
        marginBottom: `${mapHeight + 100}px`
    }))



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