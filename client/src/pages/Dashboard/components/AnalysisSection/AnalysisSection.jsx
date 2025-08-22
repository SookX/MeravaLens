import { Box, Grid, Typography } from "@mui/material"
import ImageCard from "./components/ImageCard/ImageCard"
import DataCol from "./components/DataCol/DataCol"
import SummaryCard from "./components/SummaryCard/SummaryCard"
import { StyledBox, StyledDivider } from "./styling"


const AnalysisSection = () => {
    return (
        <StyledBox>
            <Box mb={6}>
                <StyledDivider><Typography textAlign={"center"} color="primary" variant="h2">Analysis</Typography></StyledDivider>
            </Box>
            
            <Grid container spacing={3}>
                <Grid size={{ xs: 12, md: 8 }}>
                    <ImageCard />
                </Grid>
                <Grid size={{ xs: 12, md: "grow" }}>
                    <DataCol />
                </Grid>
                <Grid size={12}>
                    <SummaryCard />
                </Grid>
            </Grid>
        </StyledBox>
    )
}

export default AnalysisSection