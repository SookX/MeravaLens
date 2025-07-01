import { Box, Divider, Grid, styled, Typography } from "@mui/material"
import ImageCard from "./components/ImageCard/ImageCard"
import DataCol from "./components/DataCol/DataCol"
import { theme } from "../../../../theme/theme"
import SummaryCard from "./components/SummaryCard/SummaryCard"


const AnalysisSection = () => {
    const StyledBox = styled(Box)(({ theme })=>({
        padding: `${theme.spacing(8)} ${theme.spacing(16)}`,
        [theme.breakpoints.down("lg")]: { padding: `${theme.spacing(8)} ${theme.spacing(8)}` },
        [theme.breakpoints.down("sm")]: { padding: `${theme.spacing(8)} ${theme.spacing(2)}` }
    }))



    const StyledDivider = styled(Divider)(({ theme })=>({
        "&::before, &::after": {
            borderColor: theme.palette.text.default
        }
    }))



    return (
        <StyledBox id="analysis">
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