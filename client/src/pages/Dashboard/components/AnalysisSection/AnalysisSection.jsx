import { Divider, Grid, styled } from "@mui/material"
import ImageCard from "./components/ImageCard/ImageCard"
import DataCol from "./components/DataCol/DataCol"
import { theme } from "../../../../theme/theme"
import SummaryCard from "./components/SummaryCard/SummaryCard"


const AnalysisSection = () => {
    const StyledGrid = styled(Grid)(({ theme })=>({
        padding: `${theme.spacing(8)} ${theme.spacing(16)}`
    }))



    const StyledDivider = styled(Divider)(({ theme })=>({
        "&::before, &::after": {
            borderColor: theme.palette.text.default
        }
    }))



    return (
        <>
            <Box mb={6}>
                <StyledDivider><Typography textAlign={"center"} color="primary" variant="h2">Analysis</Typography></StyledDivider>
                <Typography textAlign="center" variant="body1">Click anywhere on the map and get a detailed analysis - segmented satellite picture as well as the latest weather and air pollution details.</Typography>
            </Box>
            <StyledGrid container spacing={3}>
                <Grid size={8}>
                    <ImageCard />
                </Grid>
                <Grid size={"grow"}>
                    <DataCol />
                </Grid>
                <Grid size={12}>
                    <SummaryCard />
                </Grid>
                
            </StyledGrid>
        </>
    )
}

export default AnalysisSection