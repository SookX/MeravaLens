import { Box, Card, Grid, Stack, styled, Typography } from "@mui/material"
import { theme } from "../../../../theme/theme"
import { Air } from "@mui/icons-material"
import ImageCard from "./components/ImageCard/ImageCard"
import DataCol from "./components/DataCol/DataCol"


const AnalysisSection = () => {
    return (
        <Box sx={{ padding: `64px 128px` }}>
            <Grid container spacing={3}>
                <Grid size={8}>
                    <ImageCard />
                </Grid>
                <Grid size={"grow"}>
                    <DataCol />
                </Grid>
            </Grid>
        </Box>
    )
}

export default AnalysisSection