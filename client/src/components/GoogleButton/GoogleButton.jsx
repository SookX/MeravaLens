import { Stack } from "@mui/material";
import { GoogleLogin } from "@react-oauth/google";
import { useContext } from "react";
import { DataContext } from "../../context/DataContext";
import { crud } from "../../api/crud";
import { useNavigate } from "react-router-dom";

const GoogleButton = ({ setError, text = "signin_with" }) => {
    // Gets global data from the context
    const { setAccess, setRefresh, setLoading } = useContext(DataContext)



    // Navigates users to another page
    const navigate = useNavigate()



    // Google authentication logic
    const handleGoogleLoginSuccess = async (credentialResponse) => {
        setLoading(true)

        const token = credentialResponse.credential;

        const response = await crud({
            method: 'post',
            url: '/users/google-login/',
            body: {
                token
            },
        });

        if (response.status === 200) {
            localStorage.setItem('access', response.data.access);
            setAccess(response.data.access);
            localStorage.setItem('refresh', response.data.refresh);
            setRefresh(response.data.refresh);
            navigate('/dashboard');
        } else {
            setError('Google register failed.');
        }

        setLoading(false)
    };

    const handleGoogleLoginFailure = () => {
        setError('Google register failed.');
    };



    return (
        <Stack alignItems="center">
            <GoogleLogin
                onSuccess={handleGoogleLoginSuccess}
                onError={handleGoogleLoginFailure}
                size="large"
                theme="filled_black"
                logo_alignment="center"
                shape="pill"
                text={text}
            />
        </Stack>
    )
}

export default GoogleButton