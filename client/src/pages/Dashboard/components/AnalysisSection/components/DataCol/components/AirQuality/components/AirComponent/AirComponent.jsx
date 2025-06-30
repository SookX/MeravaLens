import { Grid, Stack, Typography } from "@mui/material"
import { useContext } from "react"
import { DashboardContext } from "../../../../../../../../Dashboard"

const AirComponent = ({ label = "CO", objKey = "co" }) => {
    const { airPollution } = useContext(DashboardContext)



    return (
        <Grid size={4}>
            <Stack direction="row" gap={1}>
                <Typography variant="body2">{label}:</Typography>
                <Typography variant="body1">{airPollution.list[0].components[objKey]}</Typography>
            </Stack>
        </Grid>
    )
}

export default AirComponent