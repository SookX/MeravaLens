import { Grid, styled } from "@mui/material"
import ImageCard from "./components/ImageCard/ImageCard"
import DataCol from "./components/DataCol/DataCol"
import { theme } from "../../../../theme/theme"


const AnalysisSection = () => {
    const StyledGrid = styled(Grid)(({ theme })=>({
        padding: `${theme.spacing(8)} ${theme.spacing(16)}`
    }))



    return (
        <StyledGrid container spacing={3}>
            <Grid size={8}>
                <ImageCard />
            </Grid>
            <Grid size={"grow"}>
                <DataCol />
            </Grid>
            
        </StyledGrid>
    )
}

export default AnalysisSection